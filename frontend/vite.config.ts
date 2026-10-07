import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/auth/google': 'http://localhost:8001',
      '/auth/google/callback': 'http://localhost:8001',
      '/chat': 'http://localhost:8001',
      '/documents': 'http://localhost:8001',
      '/usage': 'http://localhost:8001',
      '/answers': 'http://localhost:8001',
      '/billing': 'http://localhost:8001',
      '/health': 'http://localhost:8001',
    },
  },
})
