import subprocess                             # Used to run shell commands (docker build/run)
import pytest                                 # Pytest framework for writing tests
import platform                               # Used to detect operating system


@pytest.mark.slow                             # Marks this test as slow (can be skipped in CI)
def test_docker_container_runs():
    """
    Smoke test for Docker container.

    Skipped on Windows due to Docker Desktop subprocess limitations.
    """                                      # Docstring explaining purpose and platform limitation

    if platform.system().lower() == "windows":  # Check if tests are running on Windows
        pytest.skip("Docker smoke tests are skipped on Windows")  # Skip test on Windows OS

    # Build image
    build = subprocess.run(                   # Execute docker build command
        ["docker", "build", "-t", "heart-disease-api-test", "."],  # Build image with test tag
        check=False                           # Do not raise exception on non-zero exit code
    )

    assert build.returncode == 0, "Docker build failed"  # Fail test if image build fails

    # Run container
    run = subprocess.run(                     # Execute docker run command
        ["docker", "run", "--rm", "heart-disease-api-test"],  # Run container and auto-remove
        timeout=15,                           # Kill process if it runs longer than 15 seconds
        check=False                           # Do not raise exception automatically
    )

    assert run.returncode == 0                # Test passes only if container exits successfully
