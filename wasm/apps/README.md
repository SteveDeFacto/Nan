# NAN Wasm app catalog

Every file here is baked into the `nan-wasm-manager` image and therefore covered
by Tinfoil attestation: what runs in the enclave is exactly what was measured at
deploy. To add an app, drop its compiled `.wasm` here and add an entry to
`catalog.json`.

> **This is the attested, baked-in catalog** — the curated apps the enclave runs
> today, referenced by id (e.g. `hello`). There is a *second*, separate catalog:
> the **on-chain community app store** (`NanAppCatalog` on Base, the **Apps** tab
> on the site) where anyone publishes `wasi:http` apps addressed by IPFS CID. That
> one is open discovery, not attested — see `contracts/README.md`. Run-by-CID from
> the store is phased in; until then, deploys run from *this* baked-in catalog.

## App requirements

Apps must be **`wasi:http` components** (the format `wasmtime serve` runs). The
component exports the `wasi:http/incoming-handler` interface: it receives an HTTP
request and returns a response, like a serverless function. The manager owns the
listener and binds it to a per-tenant loopback port; the app never opens its own
socket.

Sandbox defaults (nothing to configure): no filesystem, no host environment, no
network beyond the served HTTP socket, memory capped per app via `mem_mb` in the
catalog. Peer isolation = separate Wasm sandbox + separate OS process per app.

## catalog.json

```json
{
  "apps": [
    { "id": "hello", "name": "Hello", "file": "hello.wasm",
      "description": "…", "mem_mb": 128 }
  ]
}
```

- `id`     - what the frontend/user selects; what the supervisor sends as `image`.
- `file`   - the `.wasm` in this directory.
- `mem_mb` - per-instance guest memory cap, enforced on the Wasm linear memory
             via `wasmtime -W max-memory-size` (not RLIMIT_AS — wasmtime reserves
             terabytes of virtual space for bounds-checking, so an RLIMIT_AS
             small enough to bound RAM would kill the runtime). Optional;
             defaults to `WASM_APP_MEM_MB` (512).

## Building the sample `hello.wasm`

Requires the Rust toolchain and `cargo-component`:

```bash
rustup target add wasm32-wasip2
cargo install cargo-component

# in a scratch crate:
cargo component new hello --lib && cd hello
```

Set `wit/world.wit`:

```wit
package nan:hello;
world hello { export wasi:http/incoming-handler@0.2.0; }
```

`src/lib.rs`:

```rust
use wasi::http::types::{Fields, IncomingRequest, OutgoingBody, OutgoingResponse, ResponseOutparam};
wasi::http::proxy::export!(Component);
struct Component;
impl wasi::exports::http::incoming_handler::Guest for Component {
    fn handle(_req: IncomingRequest, out: ResponseOutparam) {
        let resp = OutgoingResponse::new(Fields::new());
        let body = resp.body().unwrap();
        ResponseOutparam::set(out, Ok(resp));
        let stream = body.write().unwrap();
        stream.blocking_write_and_flush(b"nan-wasm-ok\n").unwrap();
        drop(stream);
        OutgoingBody::finish(body, None).unwrap();
    }
}
```

Build and install into the catalog:

```bash
cargo component build --release --target wasm32-wasip2
cp target/wasm32-wasip2/release/hello.wasm /path/to/nan/wasm/apps/hello.wasm
```

(Exact WASI binding APIs shift between `wasi` crate versions; pin the crate
version documented for your `cargo-component`, and let the build errors guide the
handler signature. The shape above is the stable wasi:http/proxy pattern.)

## Verifying an app locally

```bash
wasmtime serve --addr 127.0.0.1:8080 apps/hello.wasm
curl 127.0.0.1:8080/    # -> nan-wasm-ok
```

If that works locally, it works in the enclave; the manager runs the identical
command per tenant on a private port.
