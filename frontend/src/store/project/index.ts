import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { ProjectState } from './state';

const defaultState: ProjectState = {
  projects: [],
};

export const projectModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
