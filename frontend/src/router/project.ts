export const projectRoutes = {
    path: 'projects/all',
    component: () => import(
      /* webpackChunkName: "main-projects" */ '@/views/main/project/Projects.vue'),
}