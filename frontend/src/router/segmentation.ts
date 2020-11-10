export const segmentationRoutes = {
    path: 'segmentations/all',
    component: () => import(
      /* webpackChunkName: "main-segmentations" */ '@/views/main/segmentation/Segmentations.vue'),
}