import os
import sys
import subprocess


def create_virtualenv(env_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    env_dir = os.path.join(script_dir, env_name)

    python_executable = sys.executable

    # Create the virtual environment
    subprocess.run([python_executable, "-m", "venv", env_dir])

    # Activate the virtual environment
    activate_script = get_activate_script(env_dir)
    activate_command = f"source {activate_script}"
    subprocess.run(activate_command, shell=True)
    print("Virtual environment created and activated!")

    # Install required packages
    install_packages()


def get_activate_script(env_name):
    if sys.platform == "win32":
        activate_script = os.path.join(env_name, "Scripts", "activate.bat")
    else:
        activate_script = os.path.join(env_name, "bin", "activate")
    return activate_script


def install_packages():
    # Add your required packages here
    # packages = ["numpy", "pandas", "matplotlib"]

    # Install each package using pip
    # for package in packages:
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

    print("Packages installed successfully!")


if __name__ == "__main__":
    env_name = input("Enter the name of the virtual environment: ")
    create_virtualenv(env_name)
