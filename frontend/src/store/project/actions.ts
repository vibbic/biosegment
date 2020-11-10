import { api } from '@/api';
import { ActionContext } from 'vuex';
import { ProjectCreate, ProjectUpdate } from '@/interfaces';
import { State } from '../state';
import { ProjectState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { commitSetProjects, commitSetProject } from './mutations';
import { dispatchCheckApiError } from '../main/actions';
import { commitAddNotification, commitRemoveNotification } from '../main/mutations';

type MainContext = ActionContext<ProjectState, State>;

export const actions = {
    async actionGetProjects(context: MainContext) {
        try {
            const response = await api.project.get(context.rootState.main.token);
            if (response) {
                commitSetProjects(context, response.data);
            }
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionUpdateProject(context: MainContext, payload: { id: number, project: ProjectUpdate }) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                api.project.update(context.rootState.main.token, payload.id, payload.project),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetProject(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Project successfully updated', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
    async actionCreateProject(context: MainContext, payload: ProjectCreate) {
        try {
            const loadingNotification = { content: 'saving', showProgress: true };
            commitAddNotification(context, loadingNotification);
            const response = (await Promise.all([
                api.project.create(context.rootState.main.token, payload),
                await new Promise((resolve, reject) => setTimeout(() => resolve(), 500)),
            ]))[0];
            commitSetProject(context, response.data);
            commitRemoveNotification(context, loadingNotification);
            commitAddNotification(context, { content: 'Project successfully created', color: 'success' });
        } catch (error) {
            await dispatchCheckApiError(context, error);
        }
    },
};

const { dispatch } = getStoreAccessors<ProjectState, State>('');

export const dispatchCreateProject = dispatch(actions.actionCreateProject);
export const dispatchGetProjects = dispatch(actions.actionGetProjects);
export const dispatchUpdateProject = dispatch(actions.actionUpdateProject);
