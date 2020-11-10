import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { AnnotationState } from './state';

const defaultState: AnnotationState = {
  annotations: [],
};

export const annotationModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
