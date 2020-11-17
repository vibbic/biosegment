import { ActionContext } from 'vuex';
import { AnnotationCreate, AnnotationUpdate } from '@/api';
import { State } from '../state';
import { AnnotationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { commitSetAnnotations, commitSetAnnotation, commitDeleteAnnotation } from './mutations';
import { dispatchCheckApiError } from '../main/actions';
import { commitAddNotification, commitRemoveNotification } from '../main/mutations';

type MainContext = ActionContext<AnnotationState, State>;

export const actions = {
    async actionGetAnnotations(context: MainContext) {
        try {
            const response = await context.rootState.main.api.annotation.readAnnotationsApiV1AnnotationsGet();
            if (response) {
                commitSetAnnotations(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionDeleteAnnotation(context: MainContext, payload: { id: number }) {
        try {
            const response = await context.rootState.main.api.annotation.deleteAnnotationApiV1AnnotationsIdDelete(payload.id);
            if (response) {
                commitDeleteAnnotation(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionUpdateAnnotation(context: MainContext, payload: { id: number, annotation: AnnotationUpdate }) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.annotation.updateAnnotationApiV1AnnotationsIdPut(payload.id, payload.annotation),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetAnnotation(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Annotation successfully updated', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionCreateAnnotation(context: MainContext, payload: AnnotationCreate) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                context.rootState.main.api.annotation.createAnnotationApiV1AnnotationsPost(payload),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetAnnotation(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Annotation successfully created', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },

};

const { dispatch } = getStoreAccessors<AnnotationState, State>('');

export const dispatchCreateAnnotation = dispatch(actions.actionCreateAnnotation);
export const dispatchGetAnnotations = dispatch(actions.actionGetAnnotations);
export const dispatchUpdateAnnotation = dispatch(actions.actionUpdateAnnotation);
export const dispatchDeleteAnnotation = dispatch(actions.actionDeleteAnnotation);
