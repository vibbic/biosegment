export const getLocalToken = () => localStorage.getItem('token');

export const saveLocalToken = (token: string) => localStorage.setItem('token', token);

export const removeLocalToken = () => localStorage.removeItem('token');

export function filterUndefined(obj) {
    Object.keys(obj).forEach((key) => obj[key] === undefined && delete obj[key]);
    return obj;
}

export function deepCopy(obj) {
    return JSON.parse(JSON.stringify(obj));
}
