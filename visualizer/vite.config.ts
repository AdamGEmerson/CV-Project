import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'
import { dirname, resolve } from 'path'
import fs from 'fs'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-clips',
      configureServer(server) {
        // Serve clips from data/clips directory
      server.middlewares.use('/clips', (req, res, next) => {
          const filename = req.url?.replace(/^\//, '') || ''
          if (!filename || filename.includes('..')) {
            res.statusCode = 400
            res.end('Invalid filename')
            return
          }
          
          const filePath = resolve(__dirname, '..', 'data', 'clips', filename)
          
          // Check if file exists
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.end('File not found')
            return
          }
          
          // Get file stats
          const stat = fs.statSync(filePath)
          const fileSize = stat.size
          const range = req.headers.range
          
          if (range) {
            // Handle range requests for video streaming
            const parts = range.replace(/bytes=/, '').split('-')
            const start = parseInt(parts[0], 10)
            const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1
            const chunksize = end - start + 1
            const file = fs.createReadStream(filePath, { start, end })
            
            res.writeHead(206, {
              'Content-Range': `bytes ${start}-${end}/${fileSize}`,
              'Accept-Ranges': 'bytes',
              'Content-Length': chunksize,
              'Content-Type': 'video/mp4'
            })
            file.pipe(res)
          } else {
            // Serve entire file
            res.writeHead(200, {
              'Content-Length': fileSize,
              'Content-Type': 'video/mp4',
              'Accept-Ranges': 'bytes'
            })
            fs.createReadStream(filePath).pipe(res)
          }
        })
      }
    },
    {
      name: 'serve-data',
      configureServer(server) {
        // Serve JSON data from data/landmarks directory
        server.middlewares.use('/data/landmarks', (req, res, next) => {
          const filename = req.url?.replace(/^\//, '') || ''
          if (!filename || filename.includes('..')) {
            res.statusCode = 400
            res.end('Invalid filename')
            return
          }
          
          const filePath = resolve(__dirname, '..', 'data', 'landmarks', filename)
          
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.end('File not found')
            return
          }
          
          const stat = fs.statSync(filePath)
          res.writeHead(200, {
            'Content-Length': stat.size,
            'Content-Type': 'application/json'
          })
          fs.createReadStream(filePath).pipe(res)
        })
      }
    }
  ],
  server: {
    fs: {
      // Allow serving files from parent directories
      allow: ['..']
    }
  }
})
