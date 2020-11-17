import { ModelState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const getters = {
    models: (state: ModelState) => state.models,
    oneModel: (state: ModelState) => (modelId: number) => {
        const filteredModels = state.models.filter((model) => model.id === modelId);
        if (filteredModels.length > 0) {
            return { ...filteredModels[0] };
        }
    },
};

const { read } = getStoreAccessors<ModelState, State>('');

export const readOneModel = read(getters.oneModel);
export const readModels = read(getters.models);
