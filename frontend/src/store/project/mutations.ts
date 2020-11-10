import { Project } from '@/interfaces';
import { ProjectState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setProjects(state: ProjectState, payload: Project[]) {
        state.projects = payload;
    },
    setProject(state: ProjectState, payload: Project) {
        const projects = state.projects.filter((project: Project) => project.id !== payload.id);
        projects.push(payload);
        state.projects = projects;
    },
};

const { commit } = getStoreAccessors<ProjectState, State>('');

export const commitSetProject = commit(mutations.setProject);
export const commitSetProjects = commit(mutations.setProjects);
