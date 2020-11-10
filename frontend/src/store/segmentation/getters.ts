import { SegmentationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const getters = {
    segmentations: (state: SegmentationState) => state.segmentations,
    oneSegmentation: (state: SegmentationState) => (segmentationId: number) => {
        const filteredSegmentations = state.segmentations.filter((segmentation) => segmentation.id === segmentationId);
        if (filteredSegmentations.length > 0) {
            return { ...filteredSegmentations[0] };
        }
    },
};

const { read } = getStoreAccessors<SegmentationState, State>('');

export const readOneSegmentation = read(getters.oneSegmentation);
export const readSegmentations = read(getters.segmentations);
