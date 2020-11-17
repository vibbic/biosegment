import RouterComponent from '@/components/RouterComponent.vue';

export const segmentationRoutes = {
    path: 'segmentations',
    component: RouterComponent,
    redirect: 'segmentations/all',
    children: [
      {
        path: 'all',
        name: 'main-segmentations-all',
        component: () => import(
          /* webpackChunkName: "main-segmentations-all" */ '@/views/main/segmentation/Segmentations.vue'),
      },
      {
        path: 'create',
        name: 'main-segmentations-create',
        component: () => import(
          /* webpackChunkName: "main-segmentations-create" */ '@/views/main/segmentation/CreateSegmentation.vue'),
      },
      {
        path: ':id/edit',
        name: 'main-segmentations-edit',
        component: () => import(
          /* webpackChunkName: "main-segmentations-edit" */ '@/views/main/segmentation/EditSegmentation.vue'),
      },
    ],
};
