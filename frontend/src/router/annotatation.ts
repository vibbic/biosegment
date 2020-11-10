export const annotationRoutes = {
    path: 'annotations/all',
    component: () => import(
      /* webpackChunkName: "main-annotations" */ '@/views/main/annotation/Annotations.vue'),
}