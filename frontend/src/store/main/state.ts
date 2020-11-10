import { API } from '@/api';
import { User } from '@/api/generator';



export interface AppNotification {
    content: string;
    color?: string;
    showProgress?: boolean;
}

export interface AppNotification {
    content: string;
    color?: string;
    showProgress?: boolean;
}

export interface MainState {
    token: string;
    api: API;
    isLoggedIn: boolean | null;
    logInError: boolean;
    userProfile: User | null;
    dashboardMiniDrawer: boolean;
    dashboardShowDrawer: boolean;
    notifications: AppNotification[];
}
