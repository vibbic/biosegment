import { mutations } from './mutations';
import { getters } from './getters';
import { actions } from './actions';
import { MainState } from './state';
import { Configuration } from '@/api';
import { createAPI } from '@/api';
import { apiUrl } from '@/env';

const defaultState: MainState = {
  isLoggedIn: null,
  token: '',
  api: createAPI(new Configuration({basePath: apiUrl,})),
  logInError: false,
  userProfile: null,
  dashboardMiniDrawer: false,
  dashboardShowDrawer: true,
  notifications: [],
};

export const mainModule = {
  state: defaultState,
  mutations,
  actions,
  getters,
};
