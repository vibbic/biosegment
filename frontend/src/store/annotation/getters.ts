import { AnnotationState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const getters = {
    annotations: (state: AnnotationState) => state.annotations,
    oneAnnotation: (state: AnnotationState) => (annotationId: number) => {
        const filteredAnnotations = state.annotations.filter((annotation) => annotation.id === annotationId);
        if (filteredAnnotations.length > 0) {
            return { ...filteredAnnotations[0] };
        }
    },
};

const { read } = getStoreAccessors<AnnotationState, State>('');

export const readOneAnnotation = read(getters.oneAnnotation);
export const readAnnotations = read(getters.annotations);
