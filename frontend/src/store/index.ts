import Vue from 'vue';
import Vuex, { StoreOptions } from 'vuex';

import { mainModule } from './main';
import { State } from './state';
import { adminModule } from './admin';
import { projectModule } from './project';
import { modelModule } from './model';
import { datasetModule } from './dataset';
import { annotationModule } from './annotation';
import { segmentationModule } from './segmentation';

Vue.use(Vuex);

const storeOptions: StoreOptions<State> = {
  modules: {
    main: mainModule,
    admin: adminModule,
    project: projectModule,
    model: modelModule,
    dataset: datasetModule,
    annotation: annotationModule,
    segmentation: segmentationModule,
  },
};

export const store = new Vuex.Store<State>(storeOptions);

export default store;
