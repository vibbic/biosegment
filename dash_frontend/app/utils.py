import os


def get_folder_list(folderName):
    files = []
    # r=root, d=directories, f = files
    for r, _, f in os.walk(folderName):
        for file in f:
            if ".png" in file:
                files.append(os.path.join(r, file))
    return files
