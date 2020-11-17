import RouterComponent from '@/components/RouterComponent.vue';

export const annotationRoutes = {
    path: 'annotations',
    component: RouterComponent,
    redirect: 'annotations/all',
    children: [
      {
        path: 'all',
        name: 'main-annotations-all',
        component: () => import(
          /* webpackChunkName: "main-annotations-all" */ '@/views/main/annotation/Annotations.vue'),
      },
      {
        path: 'create',
        name: 'main-annotations-create',
        component: () => import(
          /* webpackChunkName: "main-annotations-create" */ '@/views/main/annotation/CreateAnnotation.vue'),
      },
      {
        path: ':id/edit',
        name: 'main-annotations-edit',
        component: () => import(
          /* webpackChunkName: "main-annotations-edit" */ '@/views/main/annotation/EditAnnotation.vue'),
      },
    ],
};
