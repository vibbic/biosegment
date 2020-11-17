import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { ModelState } from './state';

const defaultState: ModelState = {
  models: [],
};

export const modelModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
