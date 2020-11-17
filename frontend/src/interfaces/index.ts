import { DatasetCreate, DatasetFileType, Resolution } from '@/api';

export function defaultResoution(): Resolution {
    return {
      x: 1,
      y: 1,
      z: 1,
    };
}


export function defaultDataset(): DatasetCreate {
    return {
      title: '',
      description: '',
      file_type: undefined,
      location: undefined,
      resolution: defaultResoution(),
      modality: undefined,
    };
}
