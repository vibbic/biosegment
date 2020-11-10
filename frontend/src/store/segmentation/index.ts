import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { SegmentationState } from './state';

const defaultState: SegmentationState = {
  segmentations: [],
};

export const segmentationModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
