import os
from subprocess import Popen, PIPE


def get_git_version():
    try:
        p = Popen(["git", "describe", "--abbrev=%d" % 4], stdout=PIPE, stderr=PIPE)
        p.stderr.close()
        line = p.stdout.readlines()[0]
        line = line.decode("utf-8").strip()
        return line
    except:
        return None


def get_text_version():
    current_folder = os.path.dirname(__file__)
    with open(os.path.join(current_folder, "version.txt")) as version_file:
        version = version_file.read().strip()
        return version


def get_version():

    version_text = get_text_version()
    version_git = get_git_version()
    if version_git is None:
        return version_text
    else:
        return version_git


def set_version():

    current_folder = os.path.dirname(__file__)
    try:
        version_git = get_git_version()
        with open(os.path.join(current_folder, "version.txt"), "w") as f:
            f.write(version_git)
            f.close()
    except:
        print("Can't read git version")


if __name__ == "__main__":
    set_version()
    print(get_version())
