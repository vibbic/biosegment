from enum import Enum


class FileType(str, Enum):
    tif2d = 'tif2d'
    tif3d = 'tif3d'
    tifseq = 'tifseq'
    pngseq = 'pngseq'

    def is_dir(self):
        if "seq" in self.value:
            return True
        return False