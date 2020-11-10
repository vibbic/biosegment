import Vue from 'vue';
import Vuex, { StoreOptions } from 'vuex';

import { mainModule } from './main';
import { State } from './state';
import { adminModule } from './admin';
import { projectModule } from './project';
import { datasetModule } from './dataset';

Vue.use(Vuex);

const storeOptions: StoreOptions<State> = {
  modules: {
    main: mainModule,
    admin: adminModule,
    project: projectModule,
    dataset: datasetModule,
  },
};

export const store = new Vuex.Store<State>(storeOptions);

export default store;
