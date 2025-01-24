import fs from 'fs';
import path from 'path';

export function readCSV(filePath: string): any[] {
    const data = fs.readFileSync(path.resolve(filePath), 'utf8');
    const lines = data.split('\n');
    const headers = lines[0].split(',');

    return lines.slice(1).map(line => {
        const values = line.split(',');
        return headers.reduce((obj, header, index) => {
            obj[header] = values[index];
            return obj;
        }, {});
    });
}
