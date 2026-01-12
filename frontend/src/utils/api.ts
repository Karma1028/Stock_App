import axios from "axios";

const api = axios.create({
    baseURL: "http://localhost:8000",
    headers: {
        "Content-Type": "application/json",
    },
});

export const fetchDashboardData = async () => {
    const { data } = await api.get("/api/dashboard");
    return data;
};

export const fetchStockDetails = async (symbol: string) => {
    const { data } = await api.get(`/api/stock/${symbol}`);
    return data;
};

export const fetchStockHistory = async (symbol: string, period: string = "1y") => {
    const { data } = await api.get(`/api/stock/${symbol}/history?period=${period}`);
    return data;
};

export const fetchNews = async () => {
    const { data } = await api.get("/api/news");
    return data;
};

export default api;
