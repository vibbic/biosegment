import { ActionContext } from 'vuex';
import { ModelCreate, ModelUpdate } from '@/api';
import { State } from '../state';
import { ModelState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { commitSetModels, commitSetModel, commitDeleteModel } from './mutations';
import { dispatchCheckApiError } from '../main/actions';
import { commitAddNotification, commitRemoveNotification } from '../main/mutations';

type MainContext = ActionContext<ModelState, State>;

export const actions = {
    async actionGetModels(context: MainContext) {
        try {
            const response = await context.rootState.main.api.model.readModelsApiV1ModelsGet();
            if (response) {
                commitSetModels(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionDeleteModel(context: MainContext, payload: { id: number }) {
        try {
            const response = await context.rootState.main.api.model.deleteModelApiV1ModelsIdDelete(payload.id);
            if (response) {
                commitDeleteModel(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionUpdateModel(context: MainContext, payload: { id: number, model: ModelUpdate }) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.model.updateModelApiV1ModelsIdPut(payload.id, payload.model),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetModel(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Model successfully updated', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionCreateModel(context: MainContext, payload: ModelCreate) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.model.createModelApiV1ModelsPost(payload),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetModel(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Model successfully created', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
};

const { dispatch } = getStoreAccessors<ModelState, State>('');

export const dispatchCreateModel = dispatch(actions.actionCreateModel);
export const dispatchGetModels = dispatch(actions.actionGetModels);
export const dispatchUpdateModel = dispatch(actions.actionUpdateModel);
export const dispatchDeleteModel = dispatch(actions.actionDeleteModel);
