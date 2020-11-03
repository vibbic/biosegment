import { ProjectState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const getters = {
    projects: (state: ProjectState) => state.projects,
    oneProject: (state: ProjectState) => (projectId: number) => {
        const filteredProjects = state.projects.filter((project) => project.id === projectId);
        if (filteredProjects.length > 0) {
            return { ...filteredProjects[0] };
        }
    },
};

const { read } = getStoreAccessors<ProjectState, State>('');

export const readOneProject = read(getters.oneProject);
export const readProjects = read(getters.projects);
