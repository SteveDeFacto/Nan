// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title NanPay — non-custodial pay-per-deploy forwarder for NAN.
/// @notice Users fund a deployment (and top it up) by paying USDC here. The
///         contract holds NOTHING: USDC is pulled from the payer and forwarded
///         to `payout` (the NAN cold wallet, e.g. nan.eth's address) in the SAME
///         transaction. It emits Paid(deploymentId, payer, amount) so the
///         supervisor — which only WATCHES this contract, never holds a key —
///         can convert each payment into runtime and extend the deployment's
///         expiry. No balances, no escrow, no operator key that can move funds.
///
/// Flow: payer approves USDC to this contract, then calls pay(deploymentId, amount).
///       USDC goes payer -> payout directly; Paid is emitted for attribution.
///       Native ETH: call payEth(deploymentId) with msg.value — same non-custodial
///       forward, PaidEth is emitted; the supervisor converts wei -> runtime at the
///       Chainlink ETH/USD rate when it sees the event.
interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract NanPay {
    address public owner;          // can update payout / hand off ownership
    address public payout;         // where USDC lands (nan.eth cold wallet)
    IERC20  public immutable usdc; // USDC token (Base: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913)

    event Paid(bytes32 indexed deploymentId, address indexed payer, uint256 amount);
    event PaidEth(bytes32 indexed deploymentId, address indexed payer, uint256 amountWei);
    event PayoutChanged(address indexed payout);
    event OwnerChanged(address indexed owner);

    constructor(address _usdc, address _payout) {
        require(_usdc != address(0) && _payout != address(0), "zero addr");
        owner = msg.sender;
        usdc = IERC20(_usdc);
        payout = _payout;
        emit PayoutChanged(_payout);
        emit OwnerChanged(msg.sender);
    }

    /// @notice Fund/top-up a deployment. USDC is forwarded straight to payout.
    /// @param deploymentId the supervisor's payment reference (keccak256 of its id)
    /// @param amount USDC amount (6 decimals)
    function pay(bytes32 deploymentId, uint256 amount) external {
        require(amount > 0, "amount=0");
        // payer -> payout directly; this contract never holds the funds
        require(usdc.transferFrom(msg.sender, payout, amount), "USDC transfer failed");
        emit Paid(deploymentId, msg.sender, amount);
    }

    /// @notice Fund/top-up a deployment with native ETH. Forwarded straight to payout
    ///         in the same tx (never held here). The supervisor prices the wei at the
    ///         Chainlink ETH/USD rate when the PaidEth event lands.
    /// @param deploymentId the supervisor's payment reference (keccak256 of its id)
    function payEth(bytes32 deploymentId) external payable {
        require(msg.value > 0, "value=0");
        (bool ok, ) = payout.call{value: msg.value}("");
        require(ok, "ETH transfer failed");
        emit PaidEth(deploymentId, msg.sender, msg.value);
    }

    function setPayout(address p) external {
        require(msg.sender == owner, "!owner");
        require(p != address(0), "zero addr");
        payout = p;
        emit PayoutChanged(p);
    }

    function setOwner(address o) external {
        require(msg.sender == owner, "!owner");
        require(o != address(0), "zero addr");
        owner = o;
        emit OwnerChanged(o);
    }
}
