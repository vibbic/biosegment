import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { DatasetState } from './state';

const defaultState: DatasetState = {
  datasets: [],
};

export const datasetModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
