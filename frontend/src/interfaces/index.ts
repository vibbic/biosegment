import { AnnotationCreate, DatasetCreate, ModelCreate, ProjectCreate, Resolution, SegmentationCreate } from '@/api';

export function defaultResolution(): Resolution {
    return {
      x: 1,
      y: 1,
      z: 1,
    };
}

export function defaultProject(): ProjectCreate {
  return {
    title: '',
    description: '',
    start_date: (new Date(Date.now())).toISOString().substr(0, 10),
    stop_date: (new Date(Date.now() + 1000 * 60 * 60 * 24 * 7)).toISOString().substr(0, 10),
  };
}

export function defaultModel(): ModelCreate {
  return {
    title: '',
    description: '',
    location: undefined,
  };
}

export function defaultDataset(): DatasetCreate {
    return {
      title: '',
      description: '',
      file_type: undefined,
      location: undefined,
      resolution: defaultResolution(),
      modality: undefined,
    };
}

export function defaultSegmentation(): SegmentationCreate {
  return {
    title: '',
    description: '',
    file_type: undefined,
    location: undefined,
  };
}

export function defaultAnnotation(): AnnotationCreate {
  return {
    title: '',
    description: '',
    file_type: undefined,
    location: undefined,
  };
}
