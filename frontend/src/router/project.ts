import RouterComponent from '@/components/RouterComponent.vue';

export const projectRoutes = {
    path: 'projects',
    component: RouterComponent,
    redirect: 'projects/all',
    children: [
      {
        path: 'all',
        name: 'main-projects-all',
        component: () => import(
          /* webpackChunkName: "main-projects-all" */ '@/views/main/project/Projects.vue'),
      },
      {
        path: 'create',
        name: 'main-projects-create',
        component: () => import(
          /* webpackChunkName: "main-projects-create" */ '@/views/main/project/CreateProject.vue'),
      },
      {
        path: ':id/edit',
        name: 'main-projects-edit',
        component: () => import(
          /* webpackChunkName: "main-projects-edit" */ '@/views/main/project/EditProject.vue'),
      },
    ],
};
