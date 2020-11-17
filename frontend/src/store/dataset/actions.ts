import { ActionContext } from 'vuex';
import { DatasetCreate, DatasetUpdate } from '@/api';
import { State } from '../state';
import { DatasetState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { commitSetDatasets, commitSetDataset, commitDeleteDataset } from './mutations';
import { dispatchCheckApiError } from '../main/actions';
import { commitAddNotification, commitRemoveNotification } from '../main/mutations';

type MainContext = ActionContext<DatasetState, State>;

export const actions = {
    async actionGetDatasets(context: MainContext) {
        try {
            const response = await context.rootState.main.api.dataset.readDatasetsApiV1DatasetsGet();
            if (response) {
                commitSetDatasets(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionDeleteDataset(context: MainContext, payload: { id: number }) {
        try {
            const response = await context.rootState.main.api.dataset.deleteDatasetApiV1DatasetsIdDelete(payload.id);
            if (response) {
                commitDeleteDataset(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionUpdateDataset(context: MainContext, payload: { id: number, dataset: DatasetUpdate }) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.dataset.updateDatasetApiV1DatasetsIdPut(payload.id, payload.dataset),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetDataset(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Dataset successfully updated', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionCreateDataset(context: MainContext, payload: DatasetCreate) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.dataset.createDatasetApiV1DatasetsPost(payload),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetDataset(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Dataset successfully created', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
};

const { dispatch } = getStoreAccessors<DatasetState, State>('');

export const dispatchCreateDataset = dispatch(actions.actionCreateDataset);
export const dispatchGetDatasets = dispatch(actions.actionGetDatasets);
export const dispatchUpdateDataset = dispatch(actions.actionUpdateDataset);
export const dispatchDeleteDataset = dispatch(actions.actionDeleteDataset);
