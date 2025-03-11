from importlib.metadata import PackageNotFoundError, version


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = "vllm"
package_version = get_version(package_name)

supported_versions = ["0.6.3"]

if package_version == "0.6.3":
    vllm_version = "0.6.3"
    from .vllm_v_0_6_3 import LLM, LLMEngine, Worker
else:
    raise ValueError(
        f"vllm version {package_version} not supported. " f"Currently supported versions are {supported_versions}."
    )
