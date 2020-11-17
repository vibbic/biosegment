import RouterComponent from '@/components/RouterComponent.vue';

export const datasetRoutes = {
    path: 'datasets',
    component: RouterComponent,
    redirect: 'datasets/all',
    children: [
      {
        path: 'all',
        name: 'main-datasets-all',
        component: () => import(
          /* webpackChunkName: "main-datasets-all" */ '@/views/main/dataset/Datasets.vue'),
      },
      {
        path: 'create',
        name: 'main-datasets-create',
        component: () => import(
          /* webpackChunkName: "main-datasets-create" */ '@/views/main/dataset/CreateDataset.vue'),
      },
    ],
};
