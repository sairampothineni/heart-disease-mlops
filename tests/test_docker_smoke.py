import subprocess
import pytest
import platform


@pytest.mark.slow
def test_docker_container_runs():
    """
    Smoke test for Docker container.

    Skipped on Windows due to Docker Desktop subprocess limitations.
    """

    if platform.system().lower() == "windows":
        pytest.skip("Docker smoke tests are skipped on Windows")

    # Build image
    build = subprocess.run(
        ["docker", "build", "-t", "heart-disease-api-test", "."],
        check=False
    )

    assert build.returncode == 0, "Docker build failed"

    # Run container
    run = subprocess.run(
        ["docker", "run", "--rm", "heart-disease-api-test"],
        timeout=15,
        check=False
    )

    assert run.returncode == 0
