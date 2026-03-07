import { Button } from '../ui/Button';
import { CAPTION_SIZES } from '../../utils/constants';

export function CaptionSizeControl({ size, onChange }) {
  return (
    <div className="flex gap-2 items-center">
      {Object.entries(CAPTION_SIZES).map(([key, label]) => (
        <Button
          key={key}
          variant={size === key ? 'primary' : 'secondary'}
          size="sm"
          onClick={() => onChange(key)}
        >
          {label}
        </Button>
      ))}
    </div>
  );
}
