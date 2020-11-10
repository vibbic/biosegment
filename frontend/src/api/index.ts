import axios from 'axios';
import { apiUrl } from '@/env';
import { User, UserUpdate, UserCreate, AnnotationsApi, DatasetsApi, ModelsApi, ProjectsApi, SegmentationsApi } from '@/interfaces';
import { Configuration } from './generator';

export interface API {
  project: ProjectsApi;
  dataset: DatasetsApi;
  model: ModelsApi;
  annotation: AnnotationsApi;
  segmentation: SegmentationsApi;
}

export function createAPI(config: Configuration) {
  return {
      project: new ProjectsApi(config),
      dataset: new DatasetsApi(config),
      model: new ModelsApi(config),
      annotation: new AnnotationsApi(config),
      segmentation: new SegmentationsApi(config),
  };
}

export function authHeaders(token: string) {
  return {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  };
}

export const api = {
  async logInGetToken(username: string, password: string) {
    const params = new URLSearchParams();
    params.append('username', username);
    params.append('password', password);

    return axios.post(`${apiUrl}/api/v1/login/access-token`, params);
  },
  async getMe(token: string) {
    return axios.get<User>(`${apiUrl}/api/v1/users/me`, authHeaders(token));
  },
  async updateMe(token: string, data: UserUpdate) {
    return axios.put<User>(`${apiUrl}/api/v1/users/me`, data, authHeaders(token));
  },
  async getUsers(token: string) {
    return axios.get<User[]>(`${apiUrl}/api/v1/users/`, authHeaders(token));
  },
  async updateUser(token: string, userId: number, data: UserUpdate) {
    return axios.put(`${apiUrl}/api/v1/users/${userId}`, data, authHeaders(token));
  },
  async createUser(token: string, data: UserCreate) {
    return axios.post(`${apiUrl}/api/v1/users/`, data, authHeaders(token));
  },
  async passwordRecovery(email: string) {
    return axios.post(`${apiUrl}/api/v1/password-recovery/${email}`);
  },
  async resetPassword(password: string, token: string) {
    return axios.post(`${apiUrl}/api/v1/reset-password/`, {
      new_password: password,
      token,
    });
  },
};
