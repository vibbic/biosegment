import axios from 'axios';
import { apiUrl } from '@/env';
import { authHeaders } from '@/api';
import { IProject, IProjectUpdate, IProjectCreate } from '@/interfaces';

export const project = {
  async get(token: string) {
    return axios.get<IProject[]>(`${apiUrl}/api/v1/projects/`, authHeaders(token));
  },
  async update(token: string, projectId: number, data: IProjectUpdate) {
    return axios.put(`${apiUrl}/api/v1/projects/${projectId}`, data, authHeaders(token));
  },
  async create(token: string, data: IProjectCreate) {
    return axios.post(`${apiUrl}/api/v1/projects/`, data, authHeaders(token));
  },
};
