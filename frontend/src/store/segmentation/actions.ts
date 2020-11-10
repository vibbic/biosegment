import { ActionContext } from 'vuex';
import { SegmentationCreate, SegmentationUpdate } from '@/interfaces';
import { State } from '../state';
import { SegmentationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { commitSetSegmentations, commitSetSegmentation, commitDeleteSegmentation } from './mutations';
import { dispatchCheckApiError } from '../main/actions';
import { commitAddNotification, commitRemoveNotification } from '../main/mutations';

type MainContext = ActionContext<SegmentationState, State>;

export const actions = {
    async actionGetSegmentations(context: MainContext) {
        try {
            const response = await context.rootState.main.api.segmentation.readSegmentationsApiV1SegmentationsGet();
            if (response) {
                commitSetSegmentations(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionDeleteSegmentation(context: MainContext, payload: { id: number }) {
        try {
            const response = await context.rootState.main.api.segmentation.deleteSegmentationApiV1SegmentationsIdDelete(payload.id);
            if (response) {
                commitDeleteSegmentation(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionUpdateSegmentation(context: MainContext, payload: { id: number, segmentation: SegmentationUpdate }) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.segmentation.updateSegmentationApiV1SegmentationsIdPut(payload.id, payload.segmentation),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetSegmentation(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Segmentation successfully updated', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionCreateSegmentation(context: MainContext, payload: SegmentationCreate) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.segmentation.createSegmentationApiV1SegmentationsPost(payload),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetSegmentation(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Segmentation successfully created', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },

};

const { dispatch } = getStoreAccessors<SegmentationState, State>('');

export const dispatchCreateSegmentation = dispatch(actions.actionCreateSegmentation);
export const dispatchGetSegmentations = dispatch(actions.actionGetSegmentations);
export const dispatchUpdateSegmentation = dispatch(actions.actionUpdateSegmentation);
export const dispatchDeleteSegmentation = dispatch(actions.actionDeleteSegmentation);
