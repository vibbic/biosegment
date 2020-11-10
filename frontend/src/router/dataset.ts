export const datasetRoutes = {
    path: 'datasets/all',
    component: () => import(
      /* webpackChunkName: "main-datasets" */ '@/views/main/dataset/Datasets.vue'),
};
