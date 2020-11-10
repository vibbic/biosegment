import axios from 'axios';
import { apiUrl } from '@/env';
import { authHeaders } from '@/api';
import { Project, ProjectUpdate, ProjectCreate } from '@/interfaces';

export const project = {
  async get(token: string) {
    return axios.get<Project[]>(`${apiUrl}/api/v1/projects/`, authHeaders(token));
  },
  async update(token: string, projectId: number, data: ProjectUpdate) {
    return axios.put(`${apiUrl}/api/v1/projects/${projectId}`, data, authHeaders(token));
  },
  async create(token: string, data: ProjectCreate) {
    return axios.post(`${apiUrl}/api/v1/projects/`, data, authHeaders(token));
  },
};
