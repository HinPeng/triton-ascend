"""New-interpreter compile worker. No wrapper invocation, launcher or device load."""
import json
import os
import sys


def compile_request(request):
    import torch
    import torch_npu  # noqa: F401
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.compiler import compile, make_backend
    from triton.runtime import driver

    from .identity import variant_key
    from .protocol import environment, publish_manifest, rebuild_source
    from .snapshot import decode

    torch.npu.set_device(request["environment"]["device"])
    target = driver.active.get_current_target()
    if environment(request["environment"]["device"], target) != request["environment"]:
        raise ValueError("worker environment mismatch")
    source = rebuild_source(request)
    backend = make_backend(target)
    options = backend.parse_options(decode(request["options"]))
    env = get_cache_invalidating_env_vars()
    if variant_key(source, backend, options, env) != request["variant_key"]:
        raise ValueError("worker VariantKey differs")
    kernel = compile(source, target=target, options=options.__dict__, _env_vars=env)
    return publish_manifest(request, kernel)


def main():
    # One persistent worker, one complete JSON request per line. Stdout belongs to
    # toolchain diagnostics; replies use a dedicated inherited pipe descriptor.
    with os.fdopen(int(sys.argv[1]), "w", buffering=1) as replies:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                result = {"manifest": compile_request(request)}
            except Exception as error:  # noqa: BLE001 - explicit cross-process error envelope
                result = {"error": type(error).__name__, "message": str(error)}
            replies.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
