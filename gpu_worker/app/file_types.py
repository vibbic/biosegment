from enum import Enum

# Make sure to keep in sync with backend FileType
# TODO use init https://docs.python.org/3/library/enum.html#planet
class FileType(Enum):
    tif2d = 'tif2d'
    tif3d = 'tif3d'
    tifseq = 'tifseq'
    pngseq = 'pngseq'

    def is_dir(self):
        if "seq" in self.value:
            return True
        return False