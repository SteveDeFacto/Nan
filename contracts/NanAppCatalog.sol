// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title NanAppCatalog — on-chain, versioned catalog of NAN Wasm apps.
/// @notice The public source of truth for "which Wasm apps exist, their releases,
///         and where each release's code lives." Each app is a lineage of versions;
///         each version is a `wasi:http` component whose bytes live on IPFS. The
///         catalog stores the content address (CID), never the bytes — a CID is a
///         hash of the exact wasm, so a caller fetches it from any IPFS peer and
///         verifies the bytes match independently. Discovery, not custody.
///
/// Identity & versioning:
///   - An app is identified by `appId = keccak256(publisher, slug)`. The slug is a
///     stable key inside the publisher's own namespace, so your "hello" and someone
///     else's "hello" are different apps (distinguished by publisher). Because the
///     appId embeds msg.sender, only you can ever write to your app — lineage
///     ownership is STRUCTURAL, not an access check that can be spoofed.
///   - `publishVersion` appends an immutable Version (its own CID, label, memMb).
///     Versions are append-only history; you don't edit a released version, you
///     publish a new one. A publisher can `yankVersion` a bad release (kept for
///     history, hidden by readers) and `editApp` the display metadata.
///   - Global CID uniqueness: a given wasm artifact is listed at most once across
///     the whole catalog, so a CID unambiguously maps to one app/version.
///
/// Trust model (mirrors NanRegistry: claims on-chain, verification off-chain):
///   - `verified` is an OPTIONAL owner-curated signal, set PER VERSION (you verify
///     a specific CID, and a new release starts unverified — it must be re-checked).
///     It does not gate execution; the CID does. The site can filter to verified.
///   - The phased "run-by-CID" deploy path references a chosen version as
///     `image.reference = ipfs://<cid>`.
contract NanAppCatalog {
    struct App {
        bytes32 appId;        // keccak256(publisher, slug); stable identity across versions
        address publisher;    // owns this app's lineage (only they add versions)
        string  slug;         // stable key within the publisher's namespace
        string  name;         // display name (editable)
        string  description;  // app-level blurb (editable)
        uint32  versionCount; // number of versions published
        uint64  createdAt;
        uint64  updatedAt;    // last version add or metadata edit
        bool    active;       // publisher can delist the whole app (kept for history)
    }
    struct Version {
        string  cid;          // IPFS CID of this release's .wasm (wasi:http component)
        string  version;      // label, e.g. "1.0.0"
        uint32  memMb;        // per-instance guest memory cap
        uint64  createdAt;
        bool    verified;     // owner-curated, per version (the CID is what's checked)
        bool    yanked;       // publisher pulled this release (kept for history)
    }

    uint256 private constant MAX_SLUG = 40;
    uint256 private constant MAX_NAME = 80;
    uint256 private constant MAX_DESC = 500;
    uint256 private constant MAX_VER  = 32;
    uint256 private constant MAX_CID  = 100;
    uint32  private constant MAX_MEM  = 8192;

    address public owner;                             // sets `verified`; can hand off
    bytes32[] private _appIds;                        // every app ever created
    mapping(bytes32 => App) private _apps;
    mapping(bytes32 => bool) private _exists;
    mapping(bytes32 => Version[]) private _versions;  // appId -> release history
    mapping(bytes32 => bool) private _cidUsed;        // keccak256(cid) -> already listed (global uniqueness)
    mapping(bytes32 => bool) private _verUsed;        // keccak256(appId, version) -> label taken (per-app uniqueness)

    event AppCreated(bytes32 indexed appId, address indexed publisher, string slug, string name);
    event AppEdited(bytes32 indexed appId, string name, string description);
    event VersionPublished(bytes32 indexed appId, uint256 indexed index, string version, string cid);
    event VersionVerified(bytes32 indexed appId, uint256 indexed index, bool verified);
    event VersionYanked(bytes32 indexed appId, uint256 indexed index);
    event AppActiveSet(bytes32 indexed appId, bool active);
    event OwnerChanged(address indexed owner);

    constructor() { owner = msg.sender; emit OwnerChanged(msg.sender); }

    /// @dev appId embeds the publisher, so a slug is owned per-address and cannot be squatted.
    function appIdOf(address publisher, string memory slug) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(publisher, slug));
    }

    /// @notice Publish a new version of your app `slug`, creating the app on first use.
    /// @return appId the app's stable id; index the new version's position.
    /// @dev Split into `_reserveCid` / `_touchApp` helpers so each has its own stack
    ///      frame (a single flat body hits "stack too deep" without viaIR).
    function publishVersion(
        string calldata slug,
        string calldata name,
        string calldata description,
        string calldata version,
        string calldata cid,
        uint32 memMb
    ) external returns (bytes32 appId, uint256 index) {
        require(bytes(version).length > 0 && bytes(version).length <= MAX_VER, "version length");
        require(bytes(cid).length > 0 && bytes(cid).length <= MAX_CID, "cid length");
        require(memMb > 0 && memMb <= MAX_MEM, "memMb range");

        _reserveCid(cid);
        appId = _touchApp(slug, name, description);

        // version labels are unique within an app, so `slug:version` resolves to
        // exactly one CID (deterministic human-friendly deploy references).
        {
            bytes32 vk = keccak256(abi.encodePacked(appId, version));
            require(!_verUsed[vk], "version exists");
            _verUsed[vk] = true;
        }

        Version[] storage vs = _versions[appId];
        vs.push(Version({
            cid: cid, version: version, memMb: memMb,
            createdAt: uint64(block.timestamp), verified: false, yanked: false
        }));
        index = vs.length - 1;

        App storage a = _apps[appId];
        a.versionCount = uint32(vs.length);
        a.updatedAt = uint64(block.timestamp);
        a.active = true;
        emit VersionPublished(appId, index, version, cid);
    }

    /// @dev Enforce global CID uniqueness (a wasm artifact is listed at most once).
    function _reserveCid(string calldata cid) private {
        bytes32 cidKey = keccak256(bytes(cid));
        require(!_cidUsed[cidKey], "cid already listed");
        _cidUsed[cidKey] = true;
    }

    /// @dev Create the app on first use, then refresh its display metadata. Returns appId.
    function _touchApp(string calldata slug, string calldata name, string calldata description)
        private
        returns (bytes32 appId)
    {
        require(bytes(slug).length > 0 && bytes(slug).length <= MAX_SLUG, "slug length");
        require(bytes(name).length > 0 && bytes(name).length <= MAX_NAME, "name length");
        require(bytes(description).length <= MAX_DESC, "desc length");
        appId = keccak256(abi.encodePacked(msg.sender, slug));
        App storage a = _apps[appId];
        if (!_exists[appId]) {
            _exists[appId] = true;
            _appIds.push(appId);
            a.appId = appId;
            a.publisher = msg.sender;
            a.slug = slug;
            a.createdAt = uint64(block.timestamp);
            emit AppCreated(appId, msg.sender, slug, name);
        }
        // the latest publish refreshes display metadata (the form pre-fills current values)
        a.name = name;
        a.description = description;
    }

    /// @notice Edit your app's display metadata without cutting a new version.
    function editApp(string calldata slug, string calldata name, string calldata description) external {
        require(bytes(name).length > 0 && bytes(name).length <= MAX_NAME, "name length");
        require(bytes(description).length <= MAX_DESC, "desc length");
        bytes32 appId = keccak256(abi.encodePacked(msg.sender, slug));
        require(_exists[appId], "unknown app");
        App storage a = _apps[appId];
        a.name = name;
        a.description = description;
        a.updatedAt = uint64(block.timestamp);
        emit AppEdited(appId, name, description);
    }

    /// @notice Delist / relist your whole app (kept on-chain for history either way).
    function setActive(string calldata slug, bool active) external {
        bytes32 appId = keccak256(abi.encodePacked(msg.sender, slug));
        require(_exists[appId], "unknown app");
        _apps[appId].active = active;
        emit AppActiveSet(appId, active);
    }

    /// @notice Pull a bad release. The version stays on-chain; readers hide it.
    function yankVersion(string calldata slug, uint256 index) external {
        bytes32 appId = keccak256(abi.encodePacked(msg.sender, slug));
        require(_exists[appId], "unknown app");
        require(index < _versions[appId].length, "bad index");
        _versions[appId][index].yanked = true;
        emit VersionYanked(appId, index);
    }

    /// @notice Owner-curated verification of a specific version (does not gate execution).
    function setVerified(bytes32 appId, uint256 index, bool v) external {
        require(msg.sender == owner, "!owner");
        require(_exists[appId], "unknown app");
        require(index < _versions[appId].length, "bad index");
        _versions[appId][index].verified = v;
        emit VersionVerified(appId, index, v);
    }

    function transferOwnership(address o) external {
        require(msg.sender == owner, "!owner");
        require(o != address(0), "zero addr");
        owner = o;
        emit OwnerChanged(o);
    }

    // ----- reads (off-chain discovery) -------------------------------------
    function appCount() external view returns (uint256) { return _appIds.length; }
    function appIdAt(uint256 i) external view returns (bytes32) { return _appIds[i]; }
    function getApp(bytes32 appId) external view returns (App memory) { return _apps[appId]; }
    function numVersions(bytes32 appId) external view returns (uint256) { return _versions[appId].length; }
    function getVersion(bytes32 appId, uint256 index) external view returns (Version memory) { return _versions[appId][index]; }

    /// @notice Paginated app list (metadata only; fetch each app's versions separately).
    function getAppsPage(uint256 start, uint256 n) external view returns (App[] memory page) {
        uint256 len = _appIds.length;
        if (start >= len) return new App[](0);
        uint256 end = start + n; if (end > len) end = len;
        page = new App[](end - start);
        for (uint256 i = start; i < end; i++) page[i - start] = _apps[_appIds[i]];
    }

    /// @notice Paginated release history for one app.
    function getVersionsPage(bytes32 appId, uint256 start, uint256 n) external view returns (Version[] memory page) {
        uint256 len = _versions[appId].length;
        if (start >= len) return new Version[](0);
        uint256 end = start + n; if (end > len) end = len;
        page = new Version[](end - start);
        for (uint256 i = start; i < end; i++) page[i - start] = _versions[appId][i];
    }
}
