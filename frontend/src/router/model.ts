import RouterComponent from '@/components/RouterComponent.vue';

export const modelRoutes = {
    path: 'models',
    component: RouterComponent,
    redirect: 'models/all',
    children: [
      {
        path: 'all',
        name: 'main-models-all',
        component: () => import(
          /* webpackChunkName: "main-models-all" */ '@/views/main/model/Models.vue'),
      },
      {
        path: 'create',
        name: 'main-models-create',
        component: () => import(
          /* webpackChunkName: "main-models-create" */ '@/views/main/model/CreateModel.vue'),
      },
      {
        path: ':id/edit',
        name: 'main-models-edit',
        component: () => import(
          /* webpackChunkName: "main-models-edit" */ '@/views/main/model/EditModel.vue'),
      },
    ],
};
