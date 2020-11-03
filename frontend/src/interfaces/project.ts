export interface IProject {
    id: number;
    title: string;
    description?: string;
    start_date?: Date;
    stop_date?: Date;
}

export interface IProjectUpdate {
    title?: string;
    description?: string;
    start_date?: Date;
    stop_date?: Date;
}

export interface IProjectCreate {
    title: string;
    description?: string;
    start_date?: Date;
    stop_date?: Date;
}
