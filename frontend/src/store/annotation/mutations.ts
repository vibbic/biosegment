import { Annotation } from '@/api';
import { AnnotationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setAnnotations(state: AnnotationState, payload: Annotation[]) {
        state.annotations = payload;
    },
    setAnnotation(state: AnnotationState, payload: Annotation) {
        const annotations = state.annotations.filter((annotation: Annotation) => annotation.id !== payload.id);
        annotations.push(payload);
        state.annotations = annotations;
    },
    deleteAnnotation(state: AnnotationState, payload: Annotation) {
        const annotations = state.annotations.filter((annotation: Annotation) => annotation.id !== payload.id);
        state.annotations = annotations;
    },
};

const { commit } = getStoreAccessors<AnnotationState, State>('');

export const commitSetAnnotation = commit(mutations.setAnnotation);
export const commitSetAnnotations = commit(mutations.setAnnotations);
export const commitDeleteAnnotation = commit(mutations.deleteAnnotation);
