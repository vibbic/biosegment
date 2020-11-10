import { Segmentation } from '@/interfaces';
import { SegmentationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setSegmentations(state: SegmentationState, payload: Segmentation[]) {
        state.segmentations = payload;
    },
    setSegmentation(state: SegmentationState, payload: Segmentation) {
        const segmentations = state.segmentations.filter((segmentation: Segmentation) => segmentation.id !== payload.id);
        segmentations.push(payload);
        state.segmentations = segmentations;
    },
    deleteSegmentation(state: SegmentationState, payload: Segmentation) {
        const segmentations = state.segmentations.filter((segmentation: Segmentation) => segmentation.id !== payload.id);
        state.segmentations = segmentations;
    },
};

const { commit } = getStoreAccessors<SegmentationState, State>('');

export const commitSetSegmentation = commit(mutations.setSegmentation);
export const commitSetSegmentations = commit(mutations.setSegmentations);
export const commitDeleteSegmentation = commit(mutations.deleteSegmentation);
