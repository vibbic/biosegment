import { User } from '@/api';
import { AdminState } from './state';
import { getStoreAccessors } from 'typesafe-vuex';
import { State } from '../state';

export const mutations = {
    setUsers(state: AdminState, payload: User[]) {
        state.users = payload;
    },
    setUser(state: AdminState, payload: User) {
        const users = state.users.filter((user: User) => user.id !== payload.id);
        users.push(payload);
        state.users = users;
    },
};

const { commit } = getStoreAccessors<AdminState, State>('');

export const commitSetUser = commit(mutations.setUser);
export const commitSetUsers = commit(mutations.setUsers);
